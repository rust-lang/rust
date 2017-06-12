// rustfmt-chain_split_single_child: true

fn main() {
    let files = fs::read_dir("tests/source").expect(
        "Couldn't read source dir",
    );
}
