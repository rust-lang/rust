pub struct Request {
    pub id: String,
    pub arg: String,
}

pub fn decode<T>() -> Result<Request, ()> {
    (|| {
        Ok(Request {
            id: "hi".to_owned(),
            arg: match Err(()) {
                Ok(v) => v,
                Err(e) => return Err(e)
            },
        })
    })()
}
