macro_rules! ciallo {
    ($($v: vis)? $name: ident) => {
    //~^ error: repetition matches empty token tree
    };
}

macro_rules! meow {
    ($name: ident $($v: vis)?) => {
    //~^ error: repetition matches empty token tree
    };
}

macro_rules! gbc {
    ($name: ident $/*
        this comment gets removed by the suggestion
        */
        ($v: vis)?) => {
    //~^ error: repetition matches empty token tree
    };
}

ciallo!(hello);

meow!(miaow, pub);

gbc!(mygo,);

fn main() {}
