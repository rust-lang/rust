#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;

extern crate std;

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

mod builtins;
mod process_pool;

use self::builtins::{cd::change_directory, status::status};

#[stem::main]
fn main(_arg: usize) -> ! {
    let mut current_path = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/"));
    let mut pool = process_pool::ProcessPool::new();

    loop {
        pool.reap_finished();

        let Some(command) = prompt() else {
            break;
        };
        let trimmed = command.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let parts = expand_pid_tokens(trimmed);
        let Some((program, args)) = parts.split_first() else {
            continue;
        };

        match program.as_str() {
            "cd" => match change_directory(&current_path, args.first().map(String::as_str)) {
                Ok(new_path) => current_path = new_path,
                Err(err) => eprintln!("cd: {}", err),
            },
            "status" => status(&current_path, &pool),
            "exit" => break,
            "background" => {
                if pool.foreground_only() {
                    println!("Enabling background operations.");
                    io::stdout().flush().ok();
                    pool.set_background();
                }
            }
            "foreground" => {
                if !pool.foreground_only() {
                    println!("Entering foreground only mode.");
                    io::stdout().flush().ok();
                    pool.set_foreground();
                }
            }
            _ => {
                if let Err(err) = pool.add(&current_path, program, args.to_vec()) {
                    println!("Error adding command to process pool: {}", err);
                    io::stdout().flush().ok();
                }
            }
        }
    }

    pool.terminate_all();
    std::process::exit(0);
}

fn prompt() -> Option<String> {
    print!(": ");
    io::stdout().flush().ok()?;

    let mut buffer = String::new();
    let stdin = io::stdin();
    let mut handle = stdin.lock();
    handle.read_line(&mut buffer).ok()?;
    Some(buffer)
}

fn expand_pid_tokens(command: &str) -> Vec<String> {
    let pid = std::process::id().to_string();
    command
        .split_whitespace()
        .map(|part| part.replace("$$", &pid))
        .collect()
}
