fn main() {
    let a = std::process::Command::new("echo")
        .arg("1")
        ,arg("2") //~ ERROR expected one of `.`, `;`, `?`, `else`, or an operator, found `,`
        .output();
}
