// something on the first line for tidy
#!/bin/bash  //~ expected `[`, found `/`

//@ reference: input.shebang

fn main() {
    println!("ok!");
}
