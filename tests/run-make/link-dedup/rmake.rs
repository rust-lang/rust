// When native libraries are passed to the linker, there used to be an annoyance
// where multiple instances of the same library in a row would cause duplication in
// outputs. This has been fixed, and this test checks that it stays fixed.
// With the --cfg flag, -ltestb gets added to the output, breaking up the chain of -ltesta.
// Without the --cfg flag, there should be a single -ltesta, no more, no less.
// See https://github.com/rust-lang/rust/pull/84794

//@ ignore-msvc

fn main() {
    rustc().input("depa.rs").run();
    rustc().input("depb.rs").run();
    rustc().input("depc.rs").run();
    let output =
        String::from_utf8(rustc().input("empty.rs").cfg("bar").command_output().stderr).unwrap();
    let pos_a1 =
        output.find("-ltesta").expect("empty.rs, compiled with --cfg, should contain -ltesta");
    let pos_b = output[pos_a1..]
        .find("-ltestb")
        .map(|pos| pos + pos_a1)
        .expect("empty.rs, compiled with --cfg, should contain -ltestb");
    let _ = output[pos_b..]
        .find("-ltesta")
        .map(|pos| pos + pos_b)
        .expect("empty.rs, compiled with --cfg, should contain a second -ltesta");
    let output = String::from_utf8(rustc().input("empty.rs").command_output().stderr).unwrap();
    assert!(output.contains("-ltesta"));
    let output = String::from_utf8(rustc().input("empty.rs").command_output().stderr).unwrap();
    assert!(!output.contains("-ltestb"));
    let output = String::from_utf8(rustc().input("empty.rs").command_output().stderr).unwrap();
    assert_eq!(output.matches("-ltesta").count, 1);
}
