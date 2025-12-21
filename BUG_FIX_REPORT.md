# CORREÇÃO DO BUG: PR #150116 - Erro "offset is not a multiple of 16" em UEFI

## 📋 Resumo Executivo

**Bug**: Compilação do target `x86_64-unknown-uefi` falha com `error: offset is not a multiple of 16`  
**Causa Raiz**: Inversão incorreta de índices de memória no PR #150116  
**Arquivo Afetado**: `compiler/rustc_abi/src/layout/coroutine.rs`  
**Status**: ✅ CORRIGIDO  

---

## 🔍 Investigação

### Timeline
- **19/12/2025**: PR #150116 mergeado no nightly  
- **20/12/2025**: Erro reportado no projeto SOBEX  
- **20/12/2025**: Causa raiz identificada e correção aplicada  

### Commit Causador
- **PR**: #150116 - "layout: Store inverse memory index in FieldsShape::Arbitrary"
- **Autor**: moulins  
- **Commit**: `b31ee3af9c67c777143c0b6707a4439c78391b69`
- **Merge**: commit `22440fd6865`  

---

## 🐛 Análise do Bug

### Contexto
O PR #150116 otimizou o armazenamento de layouts mudando:
- **ANTES**: `memory_index: IndexVec<FieldIdx, u32>` (source→memory)
- **DEPOIS**: `in_memory_order: IndexVec<u32, FieldIdx>` (memory→source)

Essa inversão elimina chamadas repetidas a `invert_bijective_mapping()`.

### O Problema
Em `compiler/rustc_abi/src/layout/coroutine.rs`, linhas 239-281, o código:

1. Cria um mapeamento intermediário para combinar campos promoted e variant
2. **ANTES DA CORREÇÃO**: Construía `combined_in_memory_order` diretamente
3. **BUG**: Preencheu como se fosse `memory_index`, mas sem inverter no final!

#### Código Bugado (linhas 250, 268)
```rust
let mut combined_in_memory_order =
    IndexVec::from_elem_n(FieldIdx::new(invalid_field_idx), invalid_field_idx);

// ...
combined_in_memory_order[memory_index] = i;  // ❌ ERRADO!
```

**Problema**: `memory_index` é do tipo `u32` (índice na memória), mas está sendo usado para indexar um array que deveria conter `FieldIdx` (índice do campo fonte).

---

## ✅ Correção Aplicada

### Mudanças em `coroutine.rs` (linhas 244-276)

#### ANTES (BUGADO):
```rust
// So instead, we build an "inverse memory_index", as if all of the
// promoted fields were being used, but leave the elements not in the
// subset as `invalid_field_idx`, which we can filter out later to
// obtain a valid (bijective) mapping.
let memory_index = in_memory_order.invert_bijective_mapping();
let invalid_field_idx = promoted_memory_index.len() + memory_index.len();
let mut combined_in_memory_order =
    IndexVec::from_elem_n(FieldIdx::new(invalid_field_idx), invalid_field_idx);

// ...
combined_in_memory_order[memory_index] = i;

// Remove the unused slots to obtain the combined `in_memory_order`
combined_in_memory_order.raw.retain(|&i| i.index() != invalid_field_idx);

variant.fields = FieldsShape::Arbitrary {
    offsets: combined_offsets,
    in_memory_order: combined_in_memory_order,
};
```

#### DEPOIS (CORRIGIDO):
```rust
// So instead, we build a "memory_index" (source→memory), as if all of the
// promoted fields were being used, but leave the elements not in the
// subset as `invalid_field_idx`, which we can filter out later to
// obtain a valid (bijective) mapping, and then invert it to get in_memory_order.
let memory_index = in_memory_order.invert_bijective_mapping();
let invalid_field_idx = promoted_memory_index.len() + memory_index.len();
let mut combined_memory_index =
    IndexVec::from_elem_n(invalid_field_idx as u32, invalid_field_idx);

// ...
combined_memory_index[i] = memory_index;  // ✅ CORRETO!

// Remove the unused slots and invert to obtain the combined `in_memory_order`
combined_memory_index.raw.retain(|&i| i as usize != invalid_field_idx);
let combined_in_memory_order = combined_memory_index.invert_bijective_mapping();

variant.fields = FieldsShape::Arbitrary {
    offsets: combined_offsets,
    in_memory_order: combined_in_memory_order,
};
```

### Diferenças Chave:
1. **Variável intermediária**: `combined_in_memory_order` → `combined_memory_index`
2. **Tipo de elemento**: `FieldIdx::new(invalid_field_idx)` → `invalid_field_idx as u32`
3. **Atribuição**: `[memory_index] = i` → `[i] = memory_index`
4. **Inversão final**: Adicionada `let combined_in_memory_order = combined_memory_index.invert_bijective_mapping();`

---

## 🎯 Por Que Isso Causava o Erro UEFI?

1. **Índices invertidos** → Campos colocados em offsets errados
2. **Offsets desalinhados** → Estruturas não respeitam alinhamento de 16 bytes
3. **Linker PE/COFF** → Detecta violação e falha com "offset is not a multiple of 16"

O formato PE/COFF (usado por UEFI) tem requisitos **muito rígidos** de alinhamento de seções. Quando os campos estão fora de ordem, o linker não consegue alinhar corretamente.

---

## 🧪 Como Testar

### 1. Compilar o compilador Rust com a correção
```bash
cd /d/rust
python x.py build --stage 1 library/std
```

### 2. Testar no projeto SOBEX
```bash
cd /d/SOBEX
cargo +stage1 build -p blk -Z build-std=core,alloc --release --target x86_64-unknown-uefi
```

### 3. Verificar sucesso
Se a compilação passar sem `error: offset is not a multiple of 16`, a correção funcionou! ✅

---

## 📤 Próximos Passos

1. ✅ Testar a correção localmente
2. ⬜ Reportar o bug upstream: https://github.com/rust-lang/rust/issues
3. ⬜ Submeter PR com a correção: https://github.com/rust-lang/rust/pull/new

### Template de Issue
```markdown
**Title**: PE/COFF alignment error after PR #150116: "offset is not a multiple of 16"

**Description**: 
After PR #150116 (commit b31ee3af9c6), compilation for `x86_64-unknown-uefi` fails with:
```
error: offset is not a multiple of 16
error: could not compile `blk` (bin "blk") due to 1 previous error
```

**Root Cause**:
In `compiler/rustc_abi/src/layout/coroutine.rs`, the code builds `combined_in_memory_order` 
directly instead of building `combined_memory_index` first and then inverting it.

**Fix**: See attached patch/PR

**Reproducer**: [link to SOBEX or minimal repro]
```

---

## 📊 Impacto

- **Targets Afetados**: Principalmente `x86_64-unknown-uefi` (pode afetar outros targets PE/COFF)
- **Severidade**: BLOQUEANTE para projetos UEFI
- **Workaround**: Usar nightly-2025-12-18 ou anterior

---

**Autor da Investigação**: Claude Code + Leona  
**Data**: 20 de dezembro de 2025  
