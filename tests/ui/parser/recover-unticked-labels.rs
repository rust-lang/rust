fn main() {
    'label: loop { break label }    //~ error: cannot find value `label` in this scope
    'label: loop { break label 0 }  //~ error: expected one of `!`, `.`, `::`, `;`, `?`, `{`, `}`, or an operator, found `0`
    'label: loop { continue label } //~ error: expected one of `.`, `;`, `?`, `}`, or an operator, found `label`
}
