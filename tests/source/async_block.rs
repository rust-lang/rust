// rustfmt-edition: Edition2018

fn main() {
    let x = async {
        Ok(())
    };
}

fn baz() {
    // test
    let x = async {
        // async blocks are great
        Ok(())
    };

    let y = async {
        Ok(())
    }; // comment
}
