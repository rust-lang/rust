use std::error::Error;
use std::io::{self, Write};
use std::process::Command;

pub fn run() -> Result<(), Box<dyn Error>> {
    let output = Command::new("pgrep").args(["-af", "qemu-system"]).output();

    let output = match output {
        Ok(o) => o,
        Err(_) => {
            println!("Failed to execute pgrep. Is it installed?");
            return Ok(());
        }
    };

    if !output.status.success() || output.stdout.is_empty() {
        println!("No QEMU instances found.");
        return Ok(());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.trim().split('\n').collect();

    if lines.is_empty() {
        println!("No QEMU instances found.");
        return Ok(());
    }

    println!("Found {} QEMU instance(s):", lines.len());
    let mut pids = Vec::new();

    for line in &lines {
        println!("  {}", line);
        if let Some(pid_str) = line.split_whitespace().next() {
            if let Ok(pid) = pid_str.parse::<u32>() {
                pids.push(pid);
            }
        }
    }

    if pids.is_empty() {
        println!("Could not parse any PIDs.");
        return Ok(());
    }

    if pids.len() == 1 {
        let pid = pids[0];
        println!("Only one instance found. Killing PID {}...", pid);
        let _ = Command::new("kill").arg(pid.to_string()).status();
        return Ok(());
    }

    print!("\nKill all? [y/N]: ");
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    if input.trim().eq_ignore_ascii_case("y") {
        for pid in pids {
            println!("Killing PID {}...", pid);
            let _ = Command::new("kill").arg(pid.to_string()).status();
        }
        println!("Done.");
    } else {
        println!("Aborted.");
    }

    Ok(())
}
