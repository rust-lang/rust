fn main() {
    format!(); //~ ERROR requires at least a format string argument
    format!("" 1); //~ ERROR expected token: `,`
    format!("", 1 1); //~ ERROR expected token: `,`
}
