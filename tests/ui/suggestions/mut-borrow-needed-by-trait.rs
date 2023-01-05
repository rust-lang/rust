use std::env::args;
use std::fs::File;
use std::io::{stdout, Write, BufWriter};

fn main() {
    let mut args = args();
    let _ = args.next();
    let dest = args.next();

    let h1; let h2; let h3;

    let fp: &dyn Write = match dest {
        Some(path) => { h1 = File::create(path).unwrap(); &h1 },
        None => { h2 = stdout(); h3 = h2.lock(); &h3 }
    };

    let fp = BufWriter::new(fp);
    //~^ ERROR the trait bound `&dyn std::io::Write: std::io::Write` is not satisfied
    //~| ERROR the trait bound `&dyn std::io::Write: std::io::Write` is not satisfied

    writeln!(fp, "hello world").unwrap(); //~ ERROR the method
}
