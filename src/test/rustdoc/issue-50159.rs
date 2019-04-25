pub trait Signal {
    type Item;
}

pub trait Signal2 {
    type Item2;
}

impl<B, C> Signal2 for B where B: Signal<Item = C> {
    type Item2 = C;
}

// @has issue_50159/struct.Switch.html
// @has - '//code' 'impl<B> Send for Switch<B> where <B as Signal>::Item: Send'
// @has - '//code' 'impl<B> Sync for Switch<B> where <B as Signal>::Item: Sync'
// @count - '//*[@id="implementations-list"]/*[@class="impl"]' 0
// @count - '//*[@id="synthetic-implementations-list"]/*[@class="impl"]' 5
pub struct Switch<B: Signal> {
    pub inner: <B as Signal2>::Item2,
}
