pub struct Item<B>(pub B);

pub struct Container<B>(pub B);

impl<B> Container<B> {
    pub fn get_item(&self) -> Result<Item<impl Default + std::fmt::Debug + PartialEq + '_>, ()> {
        Ok(Item(0))
    }
}
