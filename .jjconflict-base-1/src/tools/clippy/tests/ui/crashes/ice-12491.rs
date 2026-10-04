#![warn(clippy::needless_return)]

fn main() {
    if (true) {
        // anything一些中文
        return;
        //~^ needless_return
    }
}
