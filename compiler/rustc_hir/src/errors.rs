use crate::LangItem;

pub struct LangItemError(pub LangItem);

impl ToString for LangItemError {
    fn to_string(&self) -> String {
        format!("requires `{}` lang_item", self.0.name())
    }
}
