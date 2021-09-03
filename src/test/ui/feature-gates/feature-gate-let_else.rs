fn main() {
    let Some(x) = Some(1) else { //~ ERROR `let...else` statements are unstable
        return;
    };
}
