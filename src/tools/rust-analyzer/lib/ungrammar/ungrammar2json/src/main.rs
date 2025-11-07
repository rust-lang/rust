#![allow(clippy::print_stderr, clippy::print_stdout)]
use std::{
    env,
    io::{self, Read},
    process,
};

use ungrammar::{Grammar, Rule};

fn main() {
    if let Err(err) = try_main() {
        eprintln!("{}", err);
        process::exit(101);
    }
}

fn try_main() -> io::Result<()> {
    if env::args().count() != 1 {
        eprintln!("Usage: ungrammar2json < grammar.ungram > grammar.json");
        return Ok(());
    }
    let grammar = read_stdin()?;
    let grammar = grammar
        .parse::<Grammar>()
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;

    let mut buf = String::new();
    grammar_to_json(&grammar, write_json::object(&mut buf));
    println!("{}", buf);
    Ok(())
}

fn read_stdin() -> io::Result<String> {
    let mut buf = String::new();
    io::stdin().lock().read_to_string(&mut buf)?;
    Ok(buf)
}

fn grammar_to_json(grammar: &Grammar, mut obj: write_json::Object<'_>) {
    for node in grammar.iter() {
        let node = &grammar[node];
        rule_to_json(grammar, &node.rule, obj.object(&node.name));
    }
}

fn rule_to_json(grammar: &Grammar, rule: &Rule, mut obj: write_json::Object) {
    match rule {
        Rule::Labeled { label, rule } => {
            obj.string("label", label);
            rule_to_json(grammar, rule, obj.object("rule"))
        }
        Rule::Node(node) => {
            obj.string("node", &grammar[*node].name);
        }
        Rule::Token(token) => {
            obj.string("token", &grammar[*token].name);
        }
        Rule::Seq(rules) | Rule::Alt(rules) => {
            let tag = match rule {
                Rule::Seq(_) => "seq",
                Rule::Alt(_) => "alt",
                _ => unreachable!(),
            };
            let mut array = obj.array(tag);
            for rule in rules {
                rule_to_json(grammar, rule, array.object());
            }
        }
        Rule::Opt(arg) | Rule::Rep(arg) => {
            let tag = match rule {
                Rule::Opt(_) => "opt",
                Rule::Rep(_) => "rep",
                _ => unreachable!(),
            };
            rule_to_json(grammar, arg, obj.object(tag));
        }
    }
}
