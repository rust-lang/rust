pub trait PublicTrait<T> {}

// @has issue_46380_2/struct.PublicStruct.html
pub struct PublicStruct;

// @!has - '//*[@class="impl"]' 'impl PublicTrait<PrivateStruct> for PublicStruct'
impl PublicTrait<PrivateStruct> for PublicStruct {}

struct PrivateStruct;
