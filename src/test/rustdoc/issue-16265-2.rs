// @has issue_16265_2/index.html 'source'

trait Y {}
impl Y for Option<u32> {}
