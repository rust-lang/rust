import * as cp from 'child_process';
import * as readline from 'readline';
import { OutputChannel } from 'vscode';

interface CompilationArtifact {
    fileName: string;
    name: string;
    kind: string;
    isTest: boolean;
}

export class Cargo {
    constructor(readonly rootFolder: string, readonly output: OutputChannel) { }

    private async artifactsFromArgs(cargoArgs: string[]): Promise<CompilationArtifact[]> {
        const artifacts: CompilationArtifact[] = [];

        try {
            await this.runCargo(cargoArgs,
                message => {
                    if (message.reason === 'compiler-artifact' && message.executable) {
                        const isBinary = message.target.crate_types.includes('bin');
                        const isBuildScript = message.target.kind.includes('custom-build');
                        if ((isBinary && !isBuildScript) || message.profile.test) {
                            artifacts.push({
                                fileName: message.executable,
                                name: message.target.name,
                                kind: message.target.kind[0],
                                isTest: message.profile.test
                            });
                        }
                    } else if (message.reason === 'compiler-message') {
                        this.output.append(message.message.rendered);
                    }
                },
                stderr => this.output.append(stderr),
            );
        } catch (err) {
            this.output.show(true);
            throw new Error(`Cargo invocation has failed: ${err}`);
        }

        return artifacts;
    }

    async executableFromArgs(args: readonly string[]): Promise<string> {
        const cargoArgs = [...args, "--message-format=json"];

        const artifacts = await this.artifactsFromArgs(cargoArgs);

        if (artifacts.length === 0) {
            throw new Error('No compilation artifacts');
        } else if (artifacts.length > 1) {
            throw new Error('Multiple compilation artifacts are not supported.');
        }

        return artifacts[0].fileName;
    }

    private runCargo(
        cargoArgs: string[],
        onStdoutJson: (obj: any) => void,
        onStderrString: (data: string) => void
    ): Promise<number> {
        return new Promise((resolve, reject) => {
            const cargo = cp.spawn('cargo', cargoArgs, {
                stdio: ['ignore', 'pipe', 'pipe'],
                cwd: this.rootFolder
            });

            cargo.on('error', err => reject(new Error(`could not launch cargo: ${err}`)));

            cargo.stderr.on('data', chunk => onStderrString(chunk.toString()));

            const rl = readline.createInterface({ input: cargo.stdout });
            rl.on('line', line => {
                const message = JSON.parse(line);
                onStdoutJson(message);
            });

            cargo.on('exit', (exitCode, _) => {
                if (exitCode === 0)
                    resolve(exitCode);
                else
                    reject(new Error(`exit code: ${exitCode}.`));
            });
        });
    }
}
