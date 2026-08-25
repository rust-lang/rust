//! The data that we will serialize and deserialize.

use rustc_macros::{Decodable, Encodable};
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};

#[derive(Debug, Encodable, Decodable)]
pub(crate) struct SerializedWorkProduct {
    /// node that produced the work-product
    pub id: WorkProductId,

    /// work-product data itself
    pub work_product: WorkProduct,
}
