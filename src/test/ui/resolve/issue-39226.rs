struct Handle {}

struct Something {
    handle: Handle
}

fn main() {
    let handle: Handle = Handle {};

    let s: Something = Something {
        handle: Handle
        //~^ ERROR expected value, found struct `Handle`
    };
}
