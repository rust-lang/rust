/*
 * Copyright (c) 2026 Teenygrad.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use melior::ir::r#type::RankedTensorType;
use melior::ir::{ShapedTypeLike, Type};

use crate::errors::Error;

pub fn tensor_type<'ctx>(dimensions: &[i64], element_type: Type<'ctx>) -> RankedTensorType<'ctx> {
    RankedTensorType::new(dimensions, element_type, None)
}

pub fn tensor_type_like<'ctx>(
    tensor: RankedTensorType<'ctx>,
    value: Type<'ctx>,
) -> Result<RankedTensorType<'ctx>, Error> {
    let dims = tensor
        .dims()
        .map_err(|e: melior::error::Error| Error::InvalidType { msg: e.to_string() })?;
    Ok(RankedTensorType::new(&dims, value, None))
}
