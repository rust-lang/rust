macro_rules! foo {
    () => {
        break (); //~ ERROR `break` outside of a loop or labeled block
    };
    ($e: expr) => {
        break $e; //~ ERROR `break` outside of a loop or labeled block
    };
    (stmt $s: stmt) => {
        $s
    };
    (@ $e: expr) => {
        { break $e; } //~ ERROR `break` outside of a loop or labeled block
    };
    (=> $s: stmt) => {
        { $s }
    };
}

fn main() {
    {
        foo!();
    }
    {
        foo!(());
    }
    {
        foo!(stmt break ()); //~ ERROR `break` outside of a loop or labeled block
    }
    {
        foo!(@ ());
    }
    {
        foo!(=> break ()); //~ ERROR `break` outside of a loop or labeled block
    }
    {
        macro_rules! bar {
            () => {
                break () //~ ERROR `break` outside of a loop or labeled block
            };
        }
        bar!()
    }
}
