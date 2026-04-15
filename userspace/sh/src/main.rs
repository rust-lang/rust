#![no_std]
#![no_main]

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use abi::errors::Errno;
use abi::signal::{
    SIG_IGN, SIGCONT, SIGINT, SIGTSTP, SIGTTIN, SIGTTOU, SigAction, SigSet, wexitstatus,
    wifcontinued, wifexited, wifsignaled, wifstopped, wstopsig, wtermsig,
};
use abi::syscall::vfs_flags;
use abi::termios;
use abi::types::{stdio_mode, waitpid_flags};
use stem::syscall;
use stem::syscall::{signal, vfs};

const TTY_FD: u32 = 0;

#[derive(Clone, Copy, PartialEq, Eq)]
enum ProcessState {
    Running,
    Stopped(u8),
    Exited(u8),
    Signaled(u8),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum JobState {
    Running,
    Stopped,
    Completed,
}

struct ProcessEntry {
    pid: u32,
    state: ProcessState,
}

struct Job {
    id: usize,
    pgid: u32,
    command: String,
    background: bool,
    processes: Vec<ProcessEntry>,
    state: JobState,
    last_status: Option<i32>,
    notify: bool,
}

impl Job {
    fn new(id: usize, pgid: u32, command: String, background: bool, pids: Vec<u32>) -> Self {
        let processes = pids
            .into_iter()
            .map(|pid| ProcessEntry { pid, state: ProcessState::Running })
            .collect();
        Self {
            id,
            pgid,
            command,
            background,
            processes,
            state: JobState::Running,
            last_status: None,
            notify: false,
        }
    }

    fn recompute_state(&self) -> JobState {
        let mut any_stopped = false;
        let mut any_running = false;
        let mut all_completed = true;

        for process in &self.processes {
            match process.state {
                ProcessState::Running => {
                    any_running = true;
                    all_completed = false;
                }
                ProcessState::Stopped(_) => {
                    any_stopped = true;
                    all_completed = false;
                }
                ProcessState::Exited(_) | ProcessState::Signaled(_) => {}
            }
        }

        if all_completed {
            JobState::Completed
        } else if any_running {
            JobState::Running
        } else if any_stopped {
            JobState::Stopped
        } else {
            JobState::Running
        }
    }

    fn update_process(&mut self, pid: u32, status: i32) -> bool {
        let old_state = self.state;
        for process in &mut self.processes {
            if process.pid != pid {
                continue;
            }
            process.state = if wifexited(status) {
                ProcessState::Exited(wexitstatus(status))
            } else if wifsignaled(status) {
                ProcessState::Signaled(wtermsig(status))
            } else if wifstopped(status) {
                ProcessState::Stopped(wstopsig(status))
            } else if wifcontinued(status) {
                ProcessState::Running
            } else {
                process.state
            };
            self.last_status = Some(status);
            break;
        }

        self.state = self.recompute_state();
        old_state != self.state || wifcontinued(status)
    }

    fn mark_running(&mut self) {
        for process in &mut self.processes {
            if matches!(process.state, ProcessState::Stopped(_)) {
                process.state = ProcessState::Running;
            }
        }
        self.state = JobState::Running;
        self.notify = false;
    }
}

struct Cmd<'a> {
    program: &'a str,
    args: Vec<&'a str>,
    stdin_file: Option<&'a str>,
    stdout_file: Option<&'a str>,
    stdout_append: bool,
}

impl<'a> Cmd<'a> {
    fn parse(tokens: &[&'a str]) -> Option<Self> {
        if tokens.is_empty() {
            return None;
        }

        let mut program = None;
        let mut args = Vec::new();
        let mut stdin_file = None;
        let mut stdout_file = None;
        let mut stdout_append = false;
        let mut idx = 0;

        while idx < tokens.len() {
            match tokens[idx] {
                "<" => {
                    idx += 1;
                    if idx < tokens.len() {
                        stdin_file = Some(tokens[idx]);
                    }
                }
                ">" => {
                    idx += 1;
                    if idx < tokens.len() {
                        stdout_file = Some(tokens[idx]);
                        stdout_append = false;
                    }
                }
                ">>" => {
                    idx += 1;
                    if idx < tokens.len() {
                        stdout_file = Some(tokens[idx]);
                        stdout_append = true;
                    }
                }
                token => {
                    if program.is_none() {
                        program = Some(token);
                    } else {
                        args.push(token);
                    }
                }
            }
            idx += 1;
        }

        program.map(|program| Self { program, args, stdin_file, stdout_file, stdout_append })
    }
}

enum ReadLineResult {
    Line(String),
    Interrupted,
    Eof,
}

struct Shell {
    shell_pgid: u32,
    jobs: Vec<Job>,
    next_job_id: usize,
    last_foreground_status: Option<i32>,
}

impl Shell {
    fn new() -> Self {
        let shell_pid = syscall::getpid();
        let _ = signal::setsid();
        let _ = signal::setpgid(0, 0);
        let shell_pgid = signal::getpgrp().unwrap_or(shell_pid as i32) as u32;
        let _ = vfs::tcsetpgrp(TTY_FD, shell_pgid);
        Self { shell_pgid, jobs: Vec::new(), next_job_id: 1, last_foreground_status: None }
    }

    fn reap_children(&mut self, nohang: bool) {
        let flags = (if nohang { waitpid_flags::WNOHANG } else { 0 })
            | waitpid_flags::WUNTRACED
            | waitpid_flags::WCONTINUED;

        loop {
            match syscall::waitpid(-1, flags) {
                Ok((0, _)) => break,
                Ok((pid, status)) if pid > 0 => {
                    self.handle_child_status(pid as u32, status);
                    if !nohang {
                        break;
                    }
                }
                Ok(_) => break,
                Err(Errno::ECHILD) => break,
                Err(Errno::EINTR) => continue,
                Err(_) => break,
            }
        }
    }

    fn handle_child_status(&mut self, pid: u32, status: i32) {
        let Some(idx) =
            self.jobs.iter().position(|job| job.processes.iter().any(|process| process.pid == pid))
        else {
            return;
        };

        let state_changed = self.jobs[idx].update_process(pid, status);
        if state_changed {
            self.jobs[idx].notify = true;
        }
    }

    fn print_job_notifications(&mut self) {
        let mut idx = 0;
        while idx < self.jobs.len() {
            let should_remove = if self.jobs[idx].notify && self.jobs[idx].background {
                print_job_update(&self.jobs[idx]);
                self.jobs[idx].notify = false;
                self.jobs[idx].state == JobState::Completed
            } else {
                self.jobs[idx].state == JobState::Completed && self.jobs[idx].background
            };

            if should_remove {
                self.jobs.remove(idx);
            } else {
                idx += 1;
            }
        }
    }

    fn launch_job(&mut self, cmds: &[Cmd<'_>], background: bool, command: &str) {
        match spawn_job(cmds, background) {
            Ok((pgid, pids)) => {
                let job_id = self.next_job_id;
                self.next_job_id += 1;
                self.jobs.push(Job::new(job_id, pgid, String::from(command), background, pids));

                if background {
                    let msg = format!("[{}] {}\n", job_id, pgid);
                    write_str(&msg);
                } else {
                    stem::debug!(
                        "sh: foreground handoff start job_id={} pgid={} shell_pgid={} cmd='{}'",
                        job_id,
                        pgid,
                        self.shell_pgid,
                        command
                    );
                    let _ = vfs::tcsetpgrp(TTY_FD, pgid);
                    stem::debug!(
                        "sh: foreground handoff done job_id={} pgid={} cmd='{}'",
                        job_id,
                        pgid,
                        command
                    );
                    self.wait_for_foreground_job(job_id);
                }
            }
            Err(err) => {
                let msg = format!("sh: spawn failed: {}\n", err);
                write_str(&msg);
            }
        }
    }

    fn wait_for_foreground_job(&mut self, job_id: usize) {
        loop {
            let Some(idx) = self.job_index(job_id) else {
                break;
            };
            if self.jobs[idx].state != JobState::Running {
                break;
            }
            self.reap_children(false);
        }

        let _ = vfs::tcsetpgrp(TTY_FD, self.shell_pgid);

        let Some(idx) = self.job_index(job_id) else {
            return;
        };

        self.jobs[idx].background = false;
        self.jobs[idx].notify = false;
        self.last_foreground_status = self.jobs[idx].last_status;

        match self.jobs[idx].state {
            JobState::Stopped => {
                if let Some(status) = self.jobs[idx].last_status {
                    let msg = format!("stopped by signal {}\n", wstopsig(status));
                    write_str(&msg);
                }
            }
            JobState::Completed => {
                if let Some(status) = self.jobs[idx].last_status {
                    if wifsignaled(status) {
                        let msg = format!("terminated by signal {}\n", wtermsig(status));
                        write_str(&msg);
                    }
                }
                self.jobs.remove(idx);
            }
            JobState::Running => {}
        }
    }

    fn list_jobs(&self) {
        for job in &self.jobs {
            let state = match job.state {
                JobState::Running => "running",
                JobState::Stopped => "stopped",
                JobState::Completed => "done",
            };
            let msg = format!("[{}] {} {}\n", job.id, state, job.command);
            write_str(&msg);
        }
    }

    fn resume_job(&mut self, arg: Option<&str>, foreground: bool) {
        let Some(idx) = self.resolve_job(arg) else {
            write_str("sh: job not found\n");
            return;
        };

        let pgid = self.jobs[idx].pgid;
        self.jobs[idx].background = !foreground;
        self.jobs[idx].mark_running();

        if foreground {
            let _ = vfs::tcsetpgrp(TTY_FD, pgid);
        }

        let _ = signal::kill(-(pgid as i32), SIGCONT);

        if foreground {
            let job_id = self.jobs[idx].id;
            self.wait_for_foreground_job(job_id);
        } else {
            let msg = format!("[{}] {}\n", self.jobs[idx].id, pgid);
            write_str(&msg);
        }
    }

    fn resolve_job(&self, arg: Option<&str>) -> Option<usize> {
        if let Some(id) = arg.and_then(parse_job_id) {
            return self.job_index(id);
        }

        self.jobs.iter().rposition(|job| job.state != JobState::Completed)
    }

    fn job_index(&self, job_id: usize) -> Option<usize> {
        self.jobs.iter().position(|job| job.id == job_id)
    }

    fn terminate_jobs(&mut self) {
        for job in &self.jobs {
            let _ = signal::kill(-(job.pgid as i32), SIGCONT);
            let _ = signal::kill(-(job.pgid as i32), abi::signal::SIGTERM);
        }
        self.reap_children(true);
    }
}

fn install_signal_handlers() {
    let ignore = SigAction { handler: SIG_IGN, mask: SigSet::EMPTY, flags: 0, restorer: 0 };
    let _ = signal::sigaction(SIGINT, Some(&ignore), None);
    let _ = signal::sigaction(SIGTSTP, Some(&ignore), None);
    let _ = signal::sigaction(SIGTTIN, Some(&ignore), None);
    let _ = signal::sigaction(SIGTTOU, Some(&ignore), None);
}

fn prompt() {
    let mut buf = [0u8; 256];
    match syscall::vfs_getcwd(&mut buf) {
        Ok(n) => {
            let cwd = core::str::from_utf8(&buf[..n]).unwrap_or("/");
            let out = format!("\x1B[32mthing\x1B[0m \x1B[34m{}\x1B[0m # ", cwd);
            write_str(&out);
        }
        Err(_) => write_str("\x1B[32mthing-os\x1B[0m # "),
    }
}

struct TtyModeGuard {
    original: Option<termios::Termios>,
}

impl TtyModeGuard {
    fn raw(fd: u32) -> Self {
        let mut original = termios::Termios::default();
        if vfs::tcgetattr(fd, &mut original).is_err() {
            return Self { original: None };
        }

        let mut raw = original;
        raw.c_lflag &=
            !(termios::ICANON | termios::ECHO | termios::ECHOE | termios::ECHONL | termios::ISIG);
        raw.c_cc[termios::VMIN] = 1;
        raw.c_cc[termios::VTIME] = 0;
        if vfs::tcsetattr(fd, &raw).is_err() {
            return Self { original: None };
        }

        Self { original: Some(original) }
    }

    fn is_active(&self) -> bool {
        self.original.is_some()
    }
}

impl Drop for TtyModeGuard {
    fn drop(&mut self) {
        if let Some(original) = self.original.as_ref() {
            let _ = vfs::tcsetattr(TTY_FD, original);
        }
    }
}

fn is_word_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn trailing_word_start(bytes: &[u8]) -> Option<usize> {
    if bytes.is_empty() || !is_word_byte(*bytes.last()?) {
        return None;
    }

    let mut idx = bytes.len() - 1;
    while idx > 0 && is_word_byte(bytes[idx - 1]) {
        idx -= 1;
    }
    Some(idx)
}

fn common_prefix_len(candidates: &[String]) -> usize {
    let Some(first) = candidates.first() else {
        return 0;
    };

    let first_bytes = first.as_bytes();
    let mut shared = first_bytes.len();
    for candidate in candidates.iter().skip(1) {
        let candidate_bytes = candidate.as_bytes();
        let mut idx = 0;
        while idx < shared
            && idx < candidate_bytes.len()
            && candidate_bytes[idx] == first_bytes[idx]
        {
            idx += 1;
        }
        shared = idx;
        if shared == 0 {
            break;
        }
    }
    shared
}

fn complete_from_current_dir(fragment: &str) -> Option<String> {
    if fragment.is_empty() {
        return None;
    }

    let fd = vfs::vfs_open(".", vfs_flags::O_RDONLY).ok()?;
    let mut matches = Vec::new();
    let mut buf = [0u8; 1024];

    loop {
        match vfs::vfs_readdir(fd, &mut buf) {
            Ok(0) => break,
            Ok(n) => {
                let mut offset = 0;
                while offset < n {
                    let mut end = offset;
                    while end < n && buf[end] != 0 {
                        end += 1;
                    }

                    if let Ok(name) = core::str::from_utf8(&buf[offset..end]) {
                        if !name.is_empty() && name.starts_with(fragment) {
                            matches.push(String::from(name));
                        }
                    }

                    offset = end + 1;
                }
            }
            Err(_) => break,
        }
    }

    let _ = syscall::vfs_close(fd);

    if matches.is_empty() {
        return None;
    }

    matches.sort();
    if matches.len() == 1 {
        let mut completion = matches.remove(0);
        completion.push(' ');
        return Some(completion);
    }

    let shared = common_prefix_len(&matches);
    if shared > fragment.len() {
        return Some(String::from(&matches[0][..shared]));
    }

    None
}

fn redraw_line(bytes: &[u8]) {
    write_str("\r\x1B[K");
    prompt();
    if let Ok(text) = core::str::from_utf8(bytes) {
        write_str(text);
    }
}

fn read_line() -> ReadLineResult {
    let tty_guard = TtyModeGuard::raw(TTY_FD);
    if !tty_guard.is_active() {
        let mut buf = [0u8; 512];
        let mut bytes = Vec::new();
        loop {
            match syscall::vfs_read(0, &mut buf) {
                Ok(0) => return ReadLineResult::Eof,
                Ok(n) => {
                    for &b in &buf[..n] {
                        bytes.push(b);
                        if b == b'\n' {
                            return ReadLineResult::Line(
                                String::from_utf8(bytes).unwrap_or_default(),
                            );
                        }
                    }
                }
                Err(Errno::EINTR) => return ReadLineResult::Interrupted,
                Err(_) => return ReadLineResult::Eof,
            }
        }
    }

    let _tty_guard = tty_guard;
    let mut buf = [0u8; 1];
    let mut bytes = Vec::new();

    loop {
        match syscall::vfs_read(0, &mut buf) {
            Ok(0) => return ReadLineResult::Eof,
            Ok(_) => match buf[0] {
                b'\r' | b'\n' => {
                    write_str("\n");
                    bytes.push(b'\n');
                    return ReadLineResult::Line(String::from_utf8(bytes).unwrap_or_default());
                }
                0x03 => {
                    write_str("^C\n");
                    return ReadLineResult::Interrupted;
                }
                0x04 => {
                    if bytes.is_empty() {
                        return ReadLineResult::Eof;
                    }
                }
                0x08 | 0x7f => {
                    if bytes.pop().is_some() {
                        write_str("\x08 \x08");
                    }
                }
                b'\t' => {
                    let Some(start) = trailing_word_start(&bytes) else {
                        continue;
                    };
                    let Ok(fragment) = core::str::from_utf8(&bytes[start..]) else {
                        continue;
                    };
                    let Some(completion) = complete_from_current_dir(fragment) else {
                        continue;
                    };

                    bytes.truncate(start);
                    bytes.extend_from_slice(completion.as_bytes());
                    redraw_line(&bytes);
                }
                b => {
                    bytes.push(b);
                    let _ = syscall::vfs_write(1, &buf);
                }
            },
            Err(Errno::EINTR) => return ReadLineResult::Interrupted,
            Err(_) => return ReadLineResult::Eof,
        }
    }
}

fn parse_line<'a>(line: &'a str) -> (Vec<Cmd<'a>>, bool) {
    let mut tokens: Vec<&str> = line.split_whitespace().collect();
    let background = matches!(tokens.last().copied(), Some("&"));
    if background {
        tokens.pop();
    }

    let mut segments = Vec::new();
    let mut current = Vec::new();
    for token in tokens {
        if token == "|" {
            segments.push(current);
            current = Vec::new();
        } else {
            current.push(token);
        }
    }
    segments.push(current);

    let cmds = segments.iter().filter_map(|segment| Cmd::parse(segment)).collect();
    (cmds, background)
}

fn parse_job_id(text: &str) -> Option<usize> {
    text.strip_prefix('%').unwrap_or(text).parse().ok()
}

fn open_read(path: &str) -> Result<u32, Errno> {
    vfs::vfs_open(path, vfs_flags::O_RDONLY)
}

fn open_write(path: &str, append: bool) -> Result<u32, Errno> {
    let flags = if append {
        vfs_flags::O_WRONLY | vfs_flags::O_CREAT | vfs_flags::O_APPEND
    } else {
        vfs_flags::O_WRONLY | vfs_flags::O_CREAT | vfs_flags::O_TRUNC
    };
    vfs::vfs_open(path, flags)
}

fn spawn_job(cmds: &[Cmd<'_>], background: bool) -> Result<(u32, Vec<u32>), Errno> {
    if cmds.is_empty() {
        return Err(Errno::EINVAL);
    }

    let mut pipes = Vec::new();
    for _ in 0..cmds.len().saturating_sub(1) {
        let mut pair = [0u32; 2];
        syscall::pipe(&mut pair)?;
        pipes.push(pair);
    }

    let bg_in = if background { Some(open_read("/dev/null")?) } else { None };
    let bg_out = if background { Some(open_write("/dev/null", false)?) } else { None };

    let env = BTreeMap::new();
    let mut spawned = Vec::new();
    let mut transient_fds = Vec::new();
    let mut pgid = 0u32;

    for (idx, cmd) in cmds.iter().enumerate() {
        let path = if cmd.program.starts_with('/') {
            String::from(cmd.program)
        } else {
            format!("/bin/{}", cmd.program)
        };

        let mut argv = Vec::new();
        argv.push(path.as_bytes().to_vec());
        for arg in &cmd.args {
            argv.push(arg.as_bytes().to_vec());
        }
        let argv_slices: Vec<&[u8]> = argv.iter().map(|arg| arg.as_slice()).collect();

        let stdin_mode = if idx == 0 {
            if let Some(path) = cmd.stdin_file {
                let fd = open_read(path)?;
                transient_fds.push(fd);
                stdio_mode::fd(fd)
            } else if background {
                stdio_mode::fd(bg_in.unwrap())
            } else {
                stdio_mode::INHERIT
            }
        } else {
            stdio_mode::fd(pipes[idx - 1][0])
        };

        let stdout_mode = if idx + 1 < cmds.len() {
            stdio_mode::fd(pipes[idx][1])
        } else if let Some(path) = cmd.stdout_file {
            let fd = open_write(path, cmd.stdout_append)?;
            transient_fds.push(fd);
            stdio_mode::fd(fd)
        } else if background {
            stdio_mode::fd(bg_out.unwrap())
        } else {
            stdio_mode::INHERIT
        };

        let stderr_mode =
            if background { stdio_mode::fd(bg_out.unwrap()) } else { stdio_mode::INHERIT };

        match syscall::spawn_process_ex(
            &path,
            &argv_slices,
            &env,
            stdin_mode,
            stdout_mode,
            stderr_mode,
            0,
            &[],
        ) {
            Ok(resp) => {
                let pid = resp.child_pid;
                stem::debug!(
                    "sh: spawned '{}' pid={} idx={} background={} pending_pgid={}",
                    path,
                    pid,
                    idx,
                    background,
                    pgid
                );
                if pgid == 0 {
                    pgid = pid;
                    let _ = signal::setpgid(pid as i32, pid as i32);
                    stem::debug!("sh: setpgid leader pid={} -> pgid={}", pid, pid);
                } else {
                    let _ = signal::setpgid(pid as i32, pgid as i32);
                    stem::debug!("sh: setpgid member pid={} -> pgid={}", pid, pgid);
                }
                spawned.push(pid);
            }
            Err(err) => {
                cleanup_fds(&pipes, &transient_fds, bg_in, bg_out);
                if pgid != 0 {
                    let _ = signal::kill(-(pgid as i32), abi::signal::SIGTERM);
                }
                return Err(err);
            }
        }
    }

    cleanup_fds(&pipes, &transient_fds, bg_in, bg_out);
    Ok((pgid, spawned))
}

fn cleanup_fds(pipes: &[[u32; 2]], transient_fds: &[u32], bg_in: Option<u32>, bg_out: Option<u32>) {
    for pair in pipes {
        let _ = syscall::vfs_close(pair[0]);
        let _ = syscall::vfs_close(pair[1]);
    }
    for &fd in transient_fds {
        let _ = syscall::vfs_close(fd);
    }
    if let Some(fd) = bg_in {
        let _ = syscall::vfs_close(fd);
    }
    if let Some(fd) = bg_out {
        let _ = syscall::vfs_close(fd);
    }
}

fn print_job_update(job: &Job) {
    let state = match job.state {
        JobState::Running => "continued",
        JobState::Stopped => "stopped",
        JobState::Completed => "done",
    };

    let detail = match job.last_status {
        Some(status) if wifexited(status) => format!("exit {}", wexitstatus(status)),
        Some(status) if wifsignaled(status) => format!("signal {}", wtermsig(status)),
        Some(status) if wifstopped(status) => format!("signal {}", wstopsig(status)),
        Some(status) if wifcontinued(status) => String::from("continued"),
        _ => String::new(),
    };

    let msg = if detail.is_empty() {
        format!("[{}] {} {}\n", job.id, state, job.command)
    } else {
        format!("[{}] {} ({}) {}\n", job.id, state, detail, job.command)
    };
    write_str(&msg);
}

fn write_str(s: &str) {
    let _ = syscall::vfs_write(1, s.as_bytes());
}

#[stem::main]
fn main(_arg: usize) -> ! {
    install_signal_handlers();
    write_str("thingos sh\n");

    let mut shell = Shell::new();

    loop {
        shell.reap_children(true);
        shell.print_job_notifications();

        prompt();
        let line = match read_line() {
            ReadLineResult::Line(line) => line,
            ReadLineResult::Interrupted => {
                write_str("\n");
                continue;
            }
            ReadLineResult::Eof => break,
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let (cmds, background) = parse_line(trimmed);
        if cmds.is_empty() {
            continue;
        }

        if cmds.len() == 1 {
            match cmds[0].program {
                "exit" => break,
                "cd" => {
                    let target = cmds[0].args.first().copied().unwrap_or("/");
                    if let Err(err) = syscall::vfs_chdir(target) {
                        let msg = format!("cd: {}: {}\n", target, err);
                        write_str(&msg);
                    }
                    continue;
                }
                "jobs" => {
                    shell.list_jobs();
                    continue;
                }
                "fg" => {
                    shell.resume_job(cmds[0].args.first().copied(), true);
                    continue;
                }
                "bg" => {
                    shell.resume_job(cmds[0].args.first().copied(), false);
                    continue;
                }
                _ => {}
            }
        }

        shell.launch_job(&cmds, background, trimmed);
    }

    let _ = vfs::tcsetpgrp(TTY_FD, shell.shell_pgid);
    shell.terminate_jobs();
    syscall::exit(0)
}
