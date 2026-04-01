struct ReqMsg();
struct RespMsg();

pub type TestType = fn() -> (ReqMsg, fn(RespMsg) -> ());
