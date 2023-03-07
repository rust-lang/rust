// run-rustfix
fn main() {
    let _ = 'label: 1 + 1; //~ ERROR expected `while`, `for`, `loop` or `{` after a label

    'label: match () { () => {}, }; //~ ERROR expected `while`, `for`, `loop` or `{` after a label
    'label: match () { () => break 'label, }; //~ ERROR expected `while`, `for`, `loop` or `{` after a label
    #[allow(unused_labels)]
    'label: match () { () => 'lp: loop { break 'lp 0 }, }; //~ ERROR expected `while`, `for`, `loop` or `{` after a label

    let x = 1;
    let _i = 'label: match x { //~ ERROR expected `while`, `for`, `loop` or `{` after a label
        0 => 42,
        1 if false => break 'label 17,
        1 => {
            if true {
                break 'label 13
            } else {
                break 'label 0;
            }
        }
        _ => 1,
    };

    let other = 3;
    let _val = 'label: (1, if other == 3 { break 'label (2, 3) } else { other }); //~ ERROR expected `while`, `for`, `loop` or `{` after a label
}
