{

macro_rules! if_ok(
    ($inp: expr) => (
        match $inp {
            Ok(move v) => { move v }
            Err(move e) => { return Err(e); }
        }
    )
);

}