// rustfmt-edition: 2018

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

    spawn(
        a,
        async move {
            action();
            Ok(())
        },
    );

    spawn(
        a,
        async move || {
            action();
            Ok(())
        },
    );

    spawn(
        a,
        static async || {
            action();
            Ok(())
        },
    );

    spawn(
        a,
        static async move || {
            action();
            Ok(())
        },
    );
}
