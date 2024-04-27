// rustfmt-max_width: 120
// rustfmt-comment_width: 110

impl Struct {
    fn fun() {
        let result = match <R::RequestResult as serde::Deserialize>::deserialize(&json) {
            Ok(v) => v,
            Err(e) => {
                match <R::ErrorResult as serde::Deserialize>::deserialize(&json) {
                    Ok(v) => return Err(Error::with_json(v)),
                    Err(e2) => return Err(Error::with_json(e)),
                }
            }
        };
    }
}
