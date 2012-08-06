type T = cat;

enum cat {
    howlycat,
    meowlycat
}

fn animal() -> ~str { ~"cat" }
fn talk(c: cat) -> ~str {
    match c {
      howlycat =>  { ~"howl" }
      meowlycat => { ~"meow" }
    }
}
