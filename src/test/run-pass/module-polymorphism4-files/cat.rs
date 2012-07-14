type T = cat;

enum cat {
    howlycat,
    meowlycat
}

fn animal() -> ~str { ~"cat" }
fn talk(c: cat) -> ~str {
    alt c {
      howlycat { ~"howl" }
      meowlycat { ~"meow" }
    }
}
