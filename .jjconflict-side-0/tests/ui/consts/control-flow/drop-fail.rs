//@ revisions: stock precise

#![feature(const_destruct)]
#![cfg_attr(precise, feature(const_precise_live_drops))]


struct NotConstDestruct;

impl Drop for NotConstDestruct {
    fn drop(&mut self) {}
}

// `x` is *not* always moved into the final value and may be dropped inside the initializer.
const _: Option<NotConstDestruct> = {
    let y: Option<NotConstDestruct> = None;
    let x = Some(NotConstDestruct);
    //[stock,precise]~^ ERROR destructor of

    if true {
        x
    } else {
        y
    }
};

// We only clear `NeedsDrop` if a local is moved from in entirely. This is a shortcoming of the
// existing analysis.
const _: NotConstDestruct = {
    let vec_tuple = (NotConstDestruct,);
    //[stock]~^ ERROR destructor of

    vec_tuple.0
};

// This applies to single-field enum variants as well.
const _: NotConstDestruct = {
    let x: Result<_, NotConstDestruct> = Ok(NotConstDestruct);
    //[stock]~^ ERROR destructor of

    match x {
        Ok(x) | Err(x) => x,
    }
};

const _: Option<NotConstDestruct> = {
    let mut some = Some(NotConstDestruct);
    let mut tmp = None;
    //[stock,precise]~^ ERROR destructor of

    let mut i = 0;
    while i < 10 {
        tmp = some;
        some = None;

        // We can escape the loop with `Some` still in `tmp`,
        // which would require that it be dropped at the end of the block.
        if i > 100 {
            break;
        }

        some = tmp;
        tmp = None;

        i += 1;
    }

    some
};

fn main() {}
