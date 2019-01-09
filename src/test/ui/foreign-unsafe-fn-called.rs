mod test {
    extern {
        pub fn free();
    }
}

fn main() {
    test::free();
    //~^ ERROR call to unsafe function is unsafe
}
