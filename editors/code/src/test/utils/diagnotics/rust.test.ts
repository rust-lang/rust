import * as assert from 'assert';
import * as fs from 'fs';
import * as vscode from 'vscode';

import {
    MappedRustDiagnostic,
    mapRustDiagnosticToVsCode,
    RustDiagnostic,
    SuggestionApplicability,
} from '../../../utils/diagnostics/rust';

function loadDiagnosticFixture(name: string): RustDiagnostic {
    const jsonText = fs
        .readFileSync(
            // We're actually in our JavaScript output directory, climb out
            `${__dirname}/../../../../src/test/fixtures/rust-diagnostics/${name}.json`,
        )
        .toString();

    return JSON.parse(jsonText);
}

function mapFixtureToVsCode(name: string): MappedRustDiagnostic {
    const rd = loadDiagnosticFixture(name);
    const mapResult = mapRustDiagnosticToVsCode(rd);

    if (!mapResult) {
        return assert.fail('Mapping unexpectedly failed');
    }
    return mapResult;
}

describe('mapRustDiagnosticToVsCode', () => {
    it('should map an incompatible type for trait error', () => {
        const { diagnostic, suggestedFixes } = mapFixtureToVsCode(
            'error/E0053',
        );

        assert.strictEqual(
            diagnostic.severity,
            vscode.DiagnosticSeverity.Error,
        );
        assert.strictEqual(diagnostic.source, 'rustc');
        assert.strictEqual(
            diagnostic.message,
            [
                `method \`next\` has an incompatible type for trait`,
                `expected type \`fn(&mut ty::list_iter::ListIterator<'list, M>) -> std::option::Option<&ty::Ref<M>>\``,
                `   found type \`fn(&ty::list_iter::ListIterator<'list, M>) -> std::option::Option<&'list ty::Ref<M>>\``,
            ].join('\n'),
        );
        assert.strictEqual(diagnostic.code, 'E0053');
        assert.deepStrictEqual(diagnostic.tags, []);

        // No related information
        assert.deepStrictEqual(diagnostic.relatedInformation, []);

        // There are no suggested fixes
        assert.strictEqual(suggestedFixes.length, 0);
    });

    it('should map an unused variable warning', () => {
        const { diagnostic, suggestedFixes } = mapFixtureToVsCode(
            'warning/unused_variables',
        );

        assert.strictEqual(
            diagnostic.severity,
            vscode.DiagnosticSeverity.Warning,
        );
        assert.strictEqual(
            diagnostic.message,
            [
                'unused variable: `foo`',
                '#[warn(unused_variables)] on by default',
            ].join('\n'),
        );
        assert.strictEqual(diagnostic.code, 'unused_variables');
        assert.strictEqual(diagnostic.source, 'rustc');
        assert.deepStrictEqual(diagnostic.tags, [
            vscode.DiagnosticTag.Unnecessary,
        ]);

        // No related information
        assert.deepStrictEqual(diagnostic.relatedInformation, []);

        // One suggested fix available to prefix the variable
        assert.strictEqual(suggestedFixes.length, 1);
        const [suggestedFix] = suggestedFixes;
        assert.strictEqual(
            suggestedFix.title,
            'consider prefixing with an underscore: `_foo`',
        );
        assert.strictEqual(
            suggestedFix.applicability,
            SuggestionApplicability.MachineApplicable,
        );
    });

    it('should map a wrong number of parameters error', () => {
        const { diagnostic, suggestedFixes } = mapFixtureToVsCode(
            'error/E0061',
        );

        assert.strictEqual(
            diagnostic.severity,
            vscode.DiagnosticSeverity.Error,
        );
        assert.strictEqual(
            diagnostic.message,
            [
                'this function takes 2 parameters but 3 parameters were supplied',
                'expected 2 parameters',
            ].join('\n'),
        );
        assert.strictEqual(diagnostic.code, 'E0061');
        assert.strictEqual(diagnostic.source, 'rustc');
        assert.deepStrictEqual(diagnostic.tags, []);

        // One related information for the original definition
        const relatedInformation = diagnostic.relatedInformation;
        if (!relatedInformation) {
            assert.fail('Related information unexpectedly undefined');
            return;
        }
        assert.strictEqual(relatedInformation.length, 1);
        const [related] = relatedInformation;
        assert.strictEqual(related.message, 'defined here');

        // There are no suggested fixes
        assert.strictEqual(suggestedFixes.length, 0);
    });

    it('should map a Clippy copy pass by ref warning', () => {
        const { diagnostic, suggestedFixes } = mapFixtureToVsCode(
            'clippy/trivially_copy_pass_by_ref',
        );

        assert.strictEqual(
            diagnostic.severity,
            vscode.DiagnosticSeverity.Warning,
        );
        assert.strictEqual(diagnostic.source, 'clippy');
        assert.strictEqual(
            diagnostic.message,
            [
                'this argument is passed by reference, but would be more efficient if passed by value',
                '#[warn(clippy::trivially_copy_pass_by_ref)] implied by #[warn(clippy::all)]',
                'for further information visit https://rust-lang.github.io/rust-clippy/master/index.html#trivially_copy_pass_by_ref',
            ].join('\n'),
        );
        assert.strictEqual(diagnostic.code, 'trivially_copy_pass_by_ref');
        assert.deepStrictEqual(diagnostic.tags, []);

        // One related information for the lint definition
        const relatedInformation = diagnostic.relatedInformation;
        if (!relatedInformation) {
            assert.fail('Related information unexpectedly undefined');
            return;
        }
        assert.strictEqual(relatedInformation.length, 1);
        const [related] = relatedInformation;
        assert.strictEqual(related.message, 'lint level defined here');

        // One suggested fix to pass by value
        assert.strictEqual(suggestedFixes.length, 1);
        const [suggestedFix] = suggestedFixes;
        assert.strictEqual(
            suggestedFix.title,
            'consider passing by value instead: `self`',
        );
        // Clippy does not mark this with any applicability
        assert.strictEqual(
            suggestedFix.applicability,
            SuggestionApplicability.Unspecified,
        );
    });

    it('should map a mismatched type error', () => {
        const { diagnostic, suggestedFixes } = mapFixtureToVsCode(
            'error/E0308',
        );

        assert.strictEqual(
            diagnostic.severity,
            vscode.DiagnosticSeverity.Error,
        );
        assert.strictEqual(
            diagnostic.message,
            ['mismatched types', 'expected usize, found u32'].join('\n'),
        );
        assert.strictEqual(diagnostic.code, 'E0308');
        assert.strictEqual(diagnostic.source, 'rustc');
        assert.deepStrictEqual(diagnostic.tags, []);

        // No related information
        assert.deepStrictEqual(diagnostic.relatedInformation, []);

        // There are no suggested fixes
        assert.strictEqual(suggestedFixes.length, 0);
    });

    it('should map a macro invocation location to normal file path', () => {
        const { location, diagnostic, suggestedFixes } = mapFixtureToVsCode(
            'error/E0277',
        );

        assert.strictEqual(
            diagnostic.severity,
            vscode.DiagnosticSeverity.Error,
        );
        assert.strictEqual(
            diagnostic.message,
            [
                "can't compare `{integer}` with `&str`",
                'the trait `std::cmp::PartialEq<&str>` is not implemented for `{integer}`',
            ].join('\n'),
        );
        assert.strictEqual(diagnostic.code, 'E0277');
        assert.strictEqual(diagnostic.source, 'rustc');
        assert.deepStrictEqual(diagnostic.tags, []);

        // No related information
        assert.deepStrictEqual(diagnostic.relatedInformation, []);

        // There are no suggested fixes
        assert.strictEqual(suggestedFixes.length, 0);

        // The file url should be normal file
        // Ignore the first part because it depends on vs workspace location
        assert.strictEqual(
            location.uri.path.substr(-'src/main.rs'.length),
            'src/main.rs',
        );
    });
});
