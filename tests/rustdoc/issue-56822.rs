struct Wrapper<T>(T);

trait MyTrait {
    type Output;
}

impl<'a, I, T: 'a> MyTrait for Wrapper<I>
    where I: MyTrait<Output=&'a T>
{
    type Output = T;
}

struct Inner<'a, T>(&'a T);

impl<'a, T> MyTrait for Inner<'a, T> {
    type Output = &'a T;
}

// @has issue_56822/struct.Parser.html
// @has - '//*[@id="synthetic-implementations-list"]//*[@class="impl"]//h3[@class="code-header"]' \
// "impl<'a> Send for Parser<'a>"
pub struct Parser<'a> {
    field: <Wrapper<Inner<'a, u8>> as MyTrait>::Output
}
