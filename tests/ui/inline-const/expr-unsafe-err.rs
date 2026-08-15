const unsafe fn require_unsafe() -> usize {
    1
}

fn main() {
    const {
        require_unsafe();
        //~^ ERROR [E0133]
    }
}
