#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;

extern crate dirs;

use std::env::current_dir;
use std::io::{self, BufRead, Write};

mod builtins;
mod process_pool;

use self::builtins::{cd::change_directory, status::status};

fn main() -> Result<(), Box<std::error::Error>> {
    let mut current_path = current_dir()?;
    let mut pool = process_pool::ProcessPool::new();

    loop {
        let command = prompt().unwrap();
        let mut command_parts = command.split_whitespace();
        let program = command_parts.next().unwrap_or("");

        match program {
            "cd" => {
                if let Some(new_path) = change_directory(command_parts.next()) {
                    current_path = new_path;
                }
            }
            "status" => {
                status(&current_path, &pool);
            }
            "exit" => {
                break;
            }
            "background" => {
                if pool.foreground_only() {
                    println!("Enabling background operations.");
                    io::stdout().flush().ok().expect("Could not flush stdout");
                    pool.set_background();
                }
            }
            "foreground" => {
                if !pool.foreground_only() {
                    println!("Entering foreground only mode.");
                    io::stdout().flush().ok().expect("Could not flush stdout");
                    pool.set_foreground();
                }
            }
            "" => {}
            _ => {
                if let Err(e) = pool.add(program, command_parts.collect()) {
                    println!("Error adding command to process pool: {}", e);
                    io::stdout().flush().ok().expect("Could not flush stdout");
                };
            }
        }
    }

    Ok(())
}

fn prompt() -> Option<String> {
    print!(": ");
    io::stdout().flush().ok().expect("Could not flush stdout");

    let mut buffer = String::new();
    let stdin = io::stdin();
    let mut handle = stdin.lock();

    //Read input from the user
    handle.read_line(&mut buffer).unwrap();

    buffer = buffer.replace("$$", &alloc::format!("{}", std::process::id()));

    //Trim buffer to take whitespace off of the right-side of a string
    let buffer = buffer.trim_right();

    Some(buffer.into())
}
