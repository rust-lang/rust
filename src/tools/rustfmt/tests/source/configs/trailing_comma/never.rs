// rustfmt-trailing_comma: Never
// Trailing comma

fn main() {
    let Lorem { ipsum, dolor, sit, } = amet;
    let Lorem { ipsum, dolor, sit, amet, consectetur, adipiscing } = elit;

    // #1544
    if let VrMsg::ClientReply {request_num: reply_req_num, value, ..} = msg {
        let _ = safe_assert_eq!(reply_req_num, request_num, op);
        return Ok((request_num, op, value));
    }

    // #1710
    pub struct FileInput {
        input: StringInput,
        file_name: OsString,
    }
    match len {
        Some(len) => Ok(new(self.input, self.pos + len)),
        None => Err(self),
    }
}
