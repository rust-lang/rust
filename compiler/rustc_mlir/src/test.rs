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

use melior::Context;
use melior::utility::register_all_llvm_translations;

use crate::load_all_dialects;

pub fn create_test_context() -> Context {
    let context = Context::new();

    context.attach_diagnostic_handler(|diagnostic| {
        eprintln!("{diagnostic}");
        true
    });

    load_all_dialects(&context);
    register_all_llvm_translations(&context);

    context
}
