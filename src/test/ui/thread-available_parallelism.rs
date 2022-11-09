// run-pass
// this is an ignore list on purpose so that new platforms don't miss it:
//ignore-vxworks
//ignore-redox
//ignore-l4re

// check that std::thread::available_parallelism returns a valid value

fn main() {
    std::thread::available_parallelism().expect(
        "should return a value on all platforms that are not ignored");
}
