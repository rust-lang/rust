// format with label break value.
fn main() {

'empty_block: {}

let result = 'block: {
    if foo() {
        // comment
        break 'block 1;
    }
    if bar() { /* comment */
        break 'block      2;
    }
    3
};
}