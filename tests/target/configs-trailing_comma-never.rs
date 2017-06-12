// rustfmt-trailing_comma: Never
// Trailing comma

fn main() {
    let Lorem { ipsum, dolor, sit } = amet;
    let Lorem {
        ipsum,
        dolor,
        sit,
        amet,
        consectetur,
        adipiscing
    } = elit;

    // #1544
    if let VrMsg::ClientReply {
            request_num: reply_req_num,
            value,
            ..
        } = msg
    {
        let _ = safe_assert_eq!(reply_req_num, request_num, op);
        return Ok((request_num, op, value));
    }
}
