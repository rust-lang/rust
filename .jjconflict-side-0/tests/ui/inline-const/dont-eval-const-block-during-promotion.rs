//! Test for #150464: as of #138499, trying to evaluate const blocks during constant promotion will
//! result in a query cycle, so we shouldn't do it. Evaluation can happen when trying to promote
//! integer division and array indexing, where it's necessary for the operation to succeed to be
//! able to use it in a promoted constant.
//@ revisions: pass fail
//@[pass] check-pass

use std::mem::offset_of;

struct Thing(i32);

fn main() {
    // For a temporary involving array indexing to be promoted, we evaluate the index to make sure
    // it's in-bounds. As of #150557 we treat inline constants as maybe-out-of-bounds to avoid the
    // query cycle from evaluating them. That allows this to compile:
    let x = &([0][const { 0 }] & 0);
    // Likewise, integer divisors must be nonzero. Avoiding the query cycle allows this to compile:
    let y = &(1 / const { 1 });
    // Likewise, signed integer dividends can't be the integer minimum when the divisor is -1.
    let z = &(const { 1 } / -1);
    // These temporaries are all lifetime-extended, so they don't need to be promoted for references
    // to them to be live later in the block. Generally, code with const blocks in these positions
    // should compile as long as being promoted isn't necessary for borrow-checking to succeed.
    (x, y, z);

    // A reduced example from real code (#150464): this can't be promoted since the array is a local
    // variable, but it still resulted in a query cycle because the index was evaluated for the
    // bounds-check before checking that. By not evaluating the const block, we avoid the cycle.
    // Since this doesn't rely on promotion, it should borrow-check successfully.
    let temp = [0u8];
    let _ = &(temp[const { 0usize }] & 0u8);
    // #150464 was reported because `offset_of!` started desugaring to a const block in #148151.
    let _ = &(temp[offset_of!(Thing, 0)] & 0u8);

    // Similarly, at the time #150464 was reported, the index here was evaluated before checking
    // that the indexed expression is an array. As above, this can't be promoted, but still resulted
    // in a query cycle. By not evaluating the const block, we avoid the cycle. Since this doesn't
    // rely on promotion, it should borrow-check successfully.
    let temp: &[u8] = &[0u8];
    let _ = &(temp[const { 0usize }] & 0u8);

    // By no longer promoting these temporaries, they're dropped at the ends of their respective
    // statements, so we can't refer to them thereafter. This code no longer query-cycles, but it
    // fails to borrow-check instead.
    #[cfg(fail)]
    {
        let (x, y, z);
        x = &([0][const { 0 }] & 0);
        //[fail]~^ ERROR: temporary value dropped while borrowed
        y = &(1 / const { 1 });
        //[fail]~^ ERROR: temporary value dropped while borrowed
        z = &(const { 1 } / -1);
        //[fail]~^ ERROR: temporary value dropped while borrowed
        (x, y, z);
    }

    // Sanity check: those temporaries do promote if the const blocks are removed.
    // If constant promotion is changed so that these are no longer implicitly promoted, the
    // comments on this test file should be reworded to reflect that.
    let (x, y, z);
    x = &([0][0] & 0);
    y = &(1 / 1);
    z = &(1 / -1);
    (x, y, z);
}
