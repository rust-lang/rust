//@has tester/struct.Window.html
//@count - '//*[@class="docblock scraped-example-list"]//span[@class="highlight"]' 1
//@has - '//*[@class="docblock scraped-example-list"]//span[@class="highlight"]' 'id'
//@count - '//*[@class="docblock scraped-example-list"]//span[@class="highlight focus"]' 1
//@has - '//*[@class="docblock scraped-example-list"]//span[@class="highlight focus"]' 'id'

pub struct Window {}

impl Window {
    pub fn id(&self) -> u64 {
        todo!()
    }
}
