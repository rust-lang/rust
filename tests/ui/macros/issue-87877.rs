// check-pass

macro_rules! two_items {
    () => {
        extern "C" {}
        extern "C" {}
    };
}

macro_rules! single_expr_funneler {
    ($expr:expr) => {
        $expr; // note the semicolon, it changes the statement kind during parsing
    };
}

macro_rules! single_item_funneler {
    ($item:item) => {
        $item
    };
}

fn main() {
    single_expr_funneler! { two_items! {} }
    single_item_funneler! { two_items! {} }
}
