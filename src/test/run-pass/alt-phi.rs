

tag thing { a; b; c; }

iter foo() -> int { put 10; }

fn main() {
    auto x = true;
    alt (a) {
        case (a) { x = true; for each (int i in foo()) { } }
        case (b) { x = false; }
        case (c) { x = false; }
    }
}