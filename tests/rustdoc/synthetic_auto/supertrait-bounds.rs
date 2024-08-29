// Check that we don't add bounds to synthetic auto trait impls that are
// already implied by the item (like supertrait bounds).

// In this case we don't want to add the bounds `T: Copy` and `T: 'static`
// to the auto trait impl because they're implied by the bound `T: Bound`
// on the implementor `Type`.

pub struct Type<T: Bound>(T);

//@ has supertrait_bounds/struct.Type.html
//@ has - '//*[@id="synthetic-implementations-list"]//*[@class="impl"]//h3[@class="code-header"]' \
// "impl<T> Send for Type<T>where T: Send,"

pub trait Bound: Copy + 'static {}
