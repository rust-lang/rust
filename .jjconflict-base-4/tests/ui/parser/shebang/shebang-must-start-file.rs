// something on the first line for tidy
#!/bin/bash  //~ ERROR expected `[`, found `/`

//@ reference: shebang.position

fn main() {
    println!("ok!");
}
