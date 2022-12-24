// run-pass
fn example_err(prog: &str, arg: &str) {
    println!("{}: {}", prog, arg)
}

fn exit<F>(print: F, prog: &str, arg: &str) where F: FnOnce(&str, &str) {
    print(prog, arg);
}

struct X<F> where F: FnOnce(&str, &str) {
    err: F,
}

impl<F> X<F> where F: FnOnce(&str, &str) {
    pub fn boom(self) {
        exit(self.err, "prog", "arg");
    }
}

pub fn main(){
    let val = X {
        err: example_err,
    };
    val.boom();
}
