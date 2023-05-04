use serde::{self, Deserialize};
use std::collections::HashSet;
use std::process::Command;
const TRANSLATABLE_LIMIT: usize = 902;

#[derive(Deserialize)]
struct CompilerMessage {
    message: CompilerMessageMessage,
}

#[derive(Deserialize)]
struct CompilerMessageMessage {
    spans: Vec<Span>,
}

#[derive(Deserialize)]
struct Span {
    expansion: Option<Box<Expansion>>,
    file_name: String,
    line_start: usize,
}

#[derive(Deserialize)]
struct Expansion {
    span: Span,
}

fn main() {
    let Some(x_path) = std::env::args().skip(1).next() else {
        panic!("Usage: translatable-limits <x.py path>")
    };
    let output = Command::new(x_path)
        .args(["check", "compiler", "--json-output", "--warnings", "warn"])
        .env("RUSTFLAGS_BOOTSTRAP", "-Wrustc::untranslatable_diagnostic")
        .output()
        .expect("executing rustc")
        .stdout;
    let stdout = String::from_utf8(output).expect("non-utf8 output");
    let messages = stdout
        .lines()
        .filter(|l| l.starts_with('{'))
        .filter_map(|msg| serde_json::from_str::<CompilerMessage>(&msg).ok())
        .filter_map(|msg| {
            let mut sp = msg.message.spans.get(0)?;
            if let Some(exp) = &sp.expansion {
                sp = &exp.span
            }
            Some((sp.file_name.clone(), sp.line_start))
        })
        .collect::<HashSet<_>>();

    assert!(messages.len() <= TRANSLATABLE_LIMIT, "Newly added diagnostics should be translatable");
    if messages.len() < TRANSLATABLE_LIMIT {
        println!("Limit can be lowered to {}", messages.len());
    }
}
