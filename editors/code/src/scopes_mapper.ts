import * as vscode from 'vscode'
import { TextMateRuleSettings } from './scopes'




let mappings = new Map<string, string[]>()


const defaultMapping = new Map<string, string[]>([
    ['comment', ['comment', 'comment.block', 'comment.line', 'comment.block.documentation']],
    ['string', ['string']],
    ['keyword', ['keyword']],
    ['keyword.control', ['keyword.control', 'keyword', 'keyword.other']],
    ['keyword.unsafe', ['storage.modifier', 'keyword.other', 'keyword.control', 'keyword']],
    ['function', ['entity.name.function']],
    ['parameter', ['variable.parameter']],
    ['constant', ['constant', 'variable']],
    ['type', ['entity.name.type']],
    ['builtin', ['variable.language', 'support.type', 'support.type']],
    ['text', ['string', 'string.quoted', 'string.regexp']],
    ['attribute', ['keyword']],
    ['literal', ['string', 'string.quoted', 'string.regexp']],
    ['macro', ['support.other']],
    ['variable', ['variable']],
    ['variable.mut', ['variable', 'storage.modifier']],
    ['field', ['variable.object.property', 'meta.field.declaration', 'meta.definition.property', 'variable.other',]],
    ['module', ['entity.name.section', 'entity.other']]
]
)

// Temporary exported for debugging for now. 
export function find(scope: string): string[] {
    return mappings.get(scope) || []
}

export function toRule(scope: string, intoRule: (scope: string) => TextMateRuleSettings | undefined): TextMateRuleSettings | undefined {
    return find(scope).map(intoRule).filter(rule => rule !== undefined)[0]
}


export function load() {
    const configuration = vscode.workspace
        .getConfiguration('rust-analyzer')
        .get('scopeMappings') as Map<string, string[]> | undefined
        || new Map()

    mappings = new Map([...Array.from(defaultMapping.entries()), ...Array.from(configuration.entries())])


}