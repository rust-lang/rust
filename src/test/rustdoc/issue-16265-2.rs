// @has issue_16265_2/index.html '[src]'

trait Y {}
impl Y for Option<u32>{}
