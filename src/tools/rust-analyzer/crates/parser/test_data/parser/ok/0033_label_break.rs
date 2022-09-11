// format with label break value.
fn main() {
    'empty_block: {}

    'block: {
        do_thing();
        if condition_not_met() {
            break 'block;
        }
        do_next_thing();
        if condition_not_met() {
            break 'block;
        }
        do_last_thing();
    }

    let result = 'block: {
        if foo() {
            // comment
            break 'block 1;
        }
        if bar() {
            /* comment */
            break 'block 2;
        }
        3
    };
}
