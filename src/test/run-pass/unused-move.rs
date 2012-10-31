// Issue #3878
// Issue Name: Unused move causes a crash
// Abstract: zero-fill to block after drop

fn main()
{
    let y = ~1;
    move y;
}