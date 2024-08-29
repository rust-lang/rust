pub struct Outer<T>(Inner<T>);
pub struct Inner<T>(T);

//@ has bounds/struct.Outer.html
//@ has - '//*[@id="synthetic-implementations-list"]//*[@class="impl"]//h3[@class="code-header"]' \
// "impl<T> Unpin for Outer<T>where \
//     T: for<'any> Trait<A = (), B<'any> = (), X = ()>,"

impl<T> std::marker::Unpin for Inner<T>
where
    T: for<'any> Trait<A = (), B<'any> = (), X = ()>,
{}

pub trait Trait: SuperTrait {
    type A;
    type B<'a>;
}

pub trait SuperTrait {
    type X;
}
