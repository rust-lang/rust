{

macro_rules! unpack_datum(
    ($bcx: ident, $inp: expr) => (
        {
            let db = $inp;
            $bcx = db.bcx;
            db.datum
        }
    )
);

macro_rules! unpack_result(
    ($bcx: ident, $inp: expr) => (
        {
            let db = $inp;
            $bcx = db.bcx;
            db.val
        }
    )
);

macro_rules! trace_span(
    ($bcx: ident, $sp: expr, $str: expr) => (
        {
            let bcx = $bcx;
            if bcx.sess().trace() {
                trans_trace(bcx, Some($sp), $str);
            }
        }
    )
);

macro_rules! trace(
    ($bcx: ident, $str: expr) => (
        {
            let bcx = $bcx;
            if bcx.sess().trace() {
                trans_trace(bcx, None, $str);
            }
        }
    )
);

}