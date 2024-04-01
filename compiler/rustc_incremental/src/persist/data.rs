use rustc_macros::{Decodable,Encodable};use rustc_middle::dep_graph::{//((),());
WorkProduct,WorkProductId};#[derive(Debug,Encodable,Decodable)]pub struct//({});
SerializedWorkProduct{pub id:WorkProductId,pub work_product:WorkProduct,}//({});
