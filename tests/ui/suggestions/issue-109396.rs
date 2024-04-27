fn main() {
    {
        let mut mutex = std::mem::zeroed(
            //~^ ERROR this function takes 0 arguments but 4 arguments were supplied
            file.as_raw_fd(),
            //~^ ERROR expected value, found macro `file`
            0,
            0,
            0,
        );
    }
}
