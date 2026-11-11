// Test assignment

fn main() {
    let  some_var  : Type   ;

    let  mut mutable;

    let variable = AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA::BBBBBBBBBBBBBBBBBBBBBB::CCCCCCCCCCCCCCCCCCCCCC::EEEEEE;

    variable =   LOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOONG;

    let single_line_fit =
        DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD;

    single_line_fit =   5;single_lit_fit    >>=  10;


    // #2791
    let x = 2;;;;
}

fn break_meee() {
    {
        (block_start, block_size, margin_block_start, margin_block_end) = match (block_start,
                                                                                 block_end,
                                                                                 block_size) {
                x => 1,
                _ => 2,
            };
    }
}

// #2018
pub const EXPLAIN_UNSIZED_TUPLE_COERCION: &'static str = "Unsized tuple coercion is not stable enough for use and is subject to change";
